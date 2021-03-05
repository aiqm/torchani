import scrapy

species = ['H', 'C', 'N', 'O']
formula = ''.join([x + '*' for x in species])
urltemplate = "https://webbook.nist.gov/cgi/cbook.cgi?Value={}-{}&VType=MW&Formula=" + formula + "&NoIon=on&MatchIso=on"  # noqa: E501
min_weight = 0
max_weight = 9999999


class NISTSpider(scrapy.Spider):
    name = "NIST"

    def start_requests(self):
        start_url = urltemplate.format(min_weight, max_weight)
        yield scrapy.Request(
            url=start_url,
            callback=lambda x: self.parse_range(x, min_weight, max_weight))

    def parse_range(self, response, from_, to):
        search_result_list = response.xpath('//*[@id="main"]/ol/li/a')

        # if there are 400 result, then the range is too large,
        # half the range and repeat the search
        if len(search_result_list) == 400 and to - from_ > 1:
            mid = (from_ + to) // 2
            next_url1 = urltemplate.format(from_, mid)
            next_url2 = urltemplate.format(mid, to)
            yield scrapy.Request(
                url=next_url1,
                callback=lambda x: self.parse_range(x, from_, mid))
            yield scrapy.Request(
                url=next_url2,
                callback=lambda x: self.parse_range(x, mid, to))
        elif len(search_result_list) == 400 and to - from_ == 1:
            next_url1 = urltemplate.format(from_, from_)
            next_url2 = urltemplate.format(to, to)
            yield scrapy.Request(
                url=next_url1,
                callback=lambda x: self.parse_range(x, from_, from_))
            yield scrapy.Request(
                url=next_url2,
                callback=lambda x: self.parse_range(x, to, to))
        else:
            for i in search_result_list:
                href = i.css('a::attr("href")').extract_first()
                prefix = '/cgi/cbook.cgi?ID='
                suffix = '&Units=SI'
                nist_id = href[len(prefix): len(href) - len(suffix)]
                url3d = 'https://webbook.nist.gov/cgi/cbook.cgi?Str3File=' + nist_id  # noqa: E501
                yield scrapy.Request(
                    url=url3d,
                    callback=lambda x, n=nist_id: self.parse_sdf(x, n))

    def parse_sdf(self, response, nist_id):
        if response.text:
            text = response.text
            lines = [x.strip() for x in text.split("\r\n")]
            lines = lines[3:]
            count = int(lines[0].split()[0])
            lines = lines[1: count + 1]
            lines = [x.split()[:4] for x in lines]
            # double check if it contain unexpected elements
            good = True
            atoms = []
            for x, y, z, atype in lines:
                if atype not in species:
                    good = False
                    break
                atoms.append((atype, float(x), float(y), float(z)))
            if good:
                yield {'id': nist_id, 'size': count, 'atoms': atoms}
